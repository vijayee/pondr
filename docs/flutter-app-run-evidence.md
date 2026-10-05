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
---

# The BINDING Slice's Run Evidence (the daemon supervised, the chat real)

**Date:** 2026-10-05. **Commits:** SecretAgent `50e7894` (the create-prompt's user
record); pondr: the FFI-binding commits `6e4227d`…<the binding live>. **Gates at the
land:** SecretAgent `setarch -R ctest --test-dir cmake-build-debug` **318/318**, the
full ASan suite **319/319**, the targeted shared-lib/CA_CONFIG ASan filter
(`TestClientApi|TestSaClient`) **44/44**, valgrind clean on the touched create path
(0 errors, 0 lost — under a debug-stripped test binary: this box's valgrind cannot
read today's full debuginfo, `ML_(img_get)` beyond image size, reproducible); pondr
`flutter analyze` **0 issues**, `flutter test` **183 passed + 1 skip** (the skip =
the integration live test un-gated; on the live run everything passes — 184 + the
three FFI live pins riding `./libsa_client.so`).

## prepare.sh (the dev's one command)

`flutter/daemon/prepare.sh` (`set -euo pipefail`): cmake-configures
`$SECRETAGENT_ROOT/cmake-build-debug` when its cache is missing, builds
`sa_client` + `frame-demo`, then copies:

| Artifact | Lands at | Why there |
|---|---|---|
| `libsa_client.so` | `flutter/libsa_client.so` | the loader's `./libsa_client.so` candidate (a `flutter run` from the project dir — `sa_ffi.dart`'s resolution order) |
| `libsa_client.so` | `build/linux/{debug,release}/bundle/lib/` (whatever bundles exist) | the loader's `$ExeDir/lib` candidate (a built/installed run) |
| `frame-demo` | `flutter/daemon/frame-demo` | the supervisor's spawn target — `lib/data/bindings.dart`'s `defaultDaemonBinary()` resolves it first, the `PONDR_DAEMON_BIN` env next, the PATH's bare name last |

The script exits printing the dev run's line + the optional
`export SA_LIBRARY_PATH='<abs>/flutter/libsa_client.so'` steer (the resolution
order in `resolveSaLibraryPath`: the define, the env, `$ExeDir/lib`,
`./libsa_client.so`).

## THE RUN (on-screen, XTEST-driven, frames captured)

Method: the real 4096×2560 desktop this time (the shell reaches `:1`; `xdotool`
+ `import -window`). Screens archived at `~/.cache/pondr/run-evidence/`
(`pondr-dlogin.png`, `pondr-chat.png`, `pondr-typed3.png`, `pondr-live1.png`,
`pondr-live2.png`, `pondr-mockchat2.png`).

```
cd pondr/flutter
flutter pub get
daemon/prepare.sh
flutter run --dart-define=pondr.mode=daemon -d linux
```

The boot's spawn line (the run log, verbatim):

```
[daemon/supervisor] spawning /home/victor/Workspace/src/github.com/vijayee/pondr/
    flutter/daemon/frame-demo (attempt 1 of 6)
[daemon/supervisor] out: frame-demo serve: serving sessions over unix at
    /home/victor/.cache/pondr/pondr.sock
[daemon/supervisor] out:                 model glm-5.3-flash:cloud at
    http://127.0.0.1:11434, store sa-demo-db
```

(The probe's first connect refused while the daemon booted — the supervisor's
poll rode; the settings' seeded selection (`~/.config/pondr/settings.json`) rode
the spawn's `--model` and the model-picker's chip showed `Ollama local · GLM
5.3 Flash…` — the FFI settings service reading the truth.)

The send ("Reply with exactly one short sentence about the sea.",
`pondr-typed3.png` → `pondr-live1.png`): the composer cleared, the daemon's
store committed (`out: [INFO] store: batch 'msg.append' committed
(fire-and-post)`), the USER bubble rendered FROM THE RECORD (right, glow, the
header took the 45-character rename, `1 message`), the typing indicator up.

The REAL reply (`pondr-live2.png`) within ~1 s (glm-5.3-flash:cloud through
Ollama):

> The sea stretches endlessly, whispering ancient secrets beneath its rolling waves.

`2 messages`, the typing indicator gone. The chat's full path proved live:
`--dart-define=pondr.mode=daemon` → the supervised spawn → the connect → the
send → the record fold → the bubbles.

The mock parity: `flutter run -d linux` with NO define boots MOCK
(`pondr-mockchat2.png`: the export's seeded sessions, "No model selected") —
the swap is define-gated only.

## Findings (recorded)

1. **THE CREATE-PROMPT'S MISSING USER RECORD** (found by the no-model
   integration test, fixed at the gate): a create-path prompt lived only in
   the frame's goal meta — the events channel never carried
   `msg.append {role: user}`, so the first send of every session NEVER
   rendered its user bubble. Fix: SecretAgent `50e7894` — the create path
   posts the steer-shaped durable input BEFORE `frame_start` (the store's
   FIFO: user words ahead of the first turn), pinned by
   `TestPromptCreateCommitsTheUserMessageRecord`.
2. **THE CHAT-VIEW TIMING FLAKE** ("the chips fill the composer") was NOT a
   metrics/deactivate race — the didChangeMetrics exceptions in its logs are
   cascade noise. THE ROOT: the mock's `m${Date.now()}` message ids collide
   when two messages compose in the same real millisecond (a test's
   fake-async pump lands the whole 1100-1799 ms typing delay inside one real
   millisecond; a real instant double-send the same way) → the canvas's
   duplicate-key assertion + the aborted layout's scroll-extent throw. Fix:
   the mock's monotonic id guard (`_nextMessageId`, the `m<ms>` format kept)
   + the canvas's scroll guard now also checks `hasContentDimensions`
   (`chat_page.dart`'s post-frame callback). Five clean `chat_view_test.dart`
   runs after the fix (the pre-fix rate was ~1 failure per 3 full-file runs).
3. **THE IDLE DAEMON SPINS** (OPEN, unrepaired — recorded for the owner): a
   fresh `frame-demo serve` burns **~2.2 cores while idle** (measured: 2220
   ticks/10 s wall on a fresh temp store, NO client; unchanged after a
   client's session). Four threads each ~55% CPU from boot (the pool's two
   workers park on the inject condvar correctly — the spin is elsewhere:
   four threads all named `frame-demo`, gdb unreachable under
   `yama/ptrace_scope` from this shell). The run's UI stayed responsive
   (every interaction above landed), but the box is cooked while a daemon
   lives. Next owner seam: attribute the four hot threads (perf/gdb with
   ptrace, or a sched_poll build) — the suspects are the WaveDB engine
   workers' reopen-walk pacing and the streams loop's idle cadence.

## The integration tests' shapes (`test/daemon/integration_test.dart`)

- THE NO-MODEL SHAPE (non-live; runs whenever the binaries exist): the REAL
  `frame-demo serve` on a temp store with an EMPTY model tag → the send's
  prompt starts its frame → the user's record folds → the frame fails
  `{type: control, payload: {kind: model-missing}}` → and the pin: the fold
  ignores the control record, the send does NOT error and does NOT complete
  (the typing flag stays up) and the session carries the user bubble alone —
  WHAT EXISTS today; the control record's UI mapping is the audit view's
  recorded follow-on.
- THE LIVE SHAPE (the trio gate: the .so resolves + the daemon binary exists
  + `--dart-define=pondr.live=true` or `PONDR_LIVE=1`, plus the endpoint
  check): the settings' adoption SETs the Ollama template through CA_CONFIG
  (absent members ride the wire's "" sentinel — decoded null), the send
  answers a REAL reply, the session carries both bubbles. The plan's pinned
  tag `gemma4:latest`; this box's 2.7 GB free memory cannot load it today
  (Ollama answers `model requires more system memory (9.9 GiB)` — a REAL
  verified refusal, the events trail rode the retry + `model-error-final`
  + `turn.end` records) — `PONDR_LIVE_MODEL` overrides the tag; the live
  gate passed end-to-end against `glm-5.3-flash:cloud`.
