# The Ask UX — Pondr's half of the escalation slice — Spec

**Date:** 2026-10-13
**Status:** owner-agreed shape settled in session (inline canvas card; options + free-text; mock
asks included). No open questions.
**Companion spec:** SecretAgent `docs/superpowers/specs/2026-10-06-escalation-design.md` — the
wire/engine half (CA_ASK_REPLY 16/17, the parked ask, `sa_client_ask_reply`) LANDED there
(SecretAgent origin/master d7b855a). This spec covers ONLY the Flutter/FFI consumption.

## Why this slice exists

The runtime can now ask its owner a question: the frame parks, an `ask` record rides the events
stream (`{kind, askId, question, options[], plan|null}`), and `sa_client_ask_reply(client, sid,
ask_id, decision 0|1, value)` posts the answer back (delivered = the reply entered the frame's
mailbox; the stale/async truth is events-stream records, never the ack). The Pondr app — the
runtime's desktop face behind `--dart-define=pondr.mode=daemon` — renders none of it: a parked
turn today shows as a report-less done turn and the question never reaches the user. This slice
surfaces the ask as an inline card and posts replies through the FFI op isolate.

## Decisions carried in (settled with the owner, 2026-10-13)

| Decision | Value | Authority |
|---|---|---|
| The card's shape | **Inline canvas card** — the pending ask is the active canvas's newest special row (chat-first, mockup-consistent, scroll-history-friendly); NOT a modal overlay | owner |
| Answers | **Option buttons + free-text** — the PA `Type something.` affordance; the reply op already accepts any value; a no-options ask gets the card's own reject affordance | owner |
| Mock mode | **Mock asks included** — `MockChatService` can emit a scripted plan-shaped ask through the same store path; the full card UX is exercisable with zero C code | owner |

## 1. The data flow

1. The daemon's `ask` record arrives via `supervisor.events` → `FfiChatService`'s record fold
   (`flutter/lib/data/daemon/ffi_chat_service.dart:306` `_onRecord`) — a NEW fold case:
   `type == "ask"` → parse `{kind, askId, question, options[], plan|null}` → the
   **`PendingAskStore`** (a riverpod `Notifier`, `flutter/lib/data/chat_state.dart` beside the
   other stores): one in-flight ask per sid, keyed by sid. Engine law: one parked ask per frame —
   the store holds at most one per sid and only a reply replaces it. An unparseable ask record
   renders nothing special (the standing raw-record fallback); the card simply never appears.
2. The fold's TERMINAL contract gains its case: today `turn.end` reason `error` → failed,
   otherwise done/report (`ffi_chat_service.dart:429-445`). Reason kind **`blocked`** → the turn
   is LEFT OPEN — no ChatFailed, no ChatDone. The pending-ask card IS that turn's face until the
   reply re-enters and the next terminal arrives. (Today a parked turn would render
   report-less-done.) WHILE a turn is parked, the send flow's typing indicator honestly rides —
   nothing feeds a terminal until the reply's turn terminal arrives (the parked turn is open).
3. `type == "ask.reply"` → clear that sid's store entry. The RESOLUTION record itself renders no
   canvas text: the answer's user-side `msg.append` record ALREADY lands as an ordinary bubble
   (the engine composes it in the same batch) — the conversation shows "you: Approve" naturally.
4. The reply: a card interaction → `native.askReply(sid, askId, decision, value)` → a new
   `SaOpCommand` (serial + the interrupt-shaped cb) → the op isolate calls
   `sa_client_ask_reply` → the cb resolves the Future: delivered → the card clears when the
   `ask.reply` record arrives; delivered:false / transport failure → the card STAYS + a small
   error line ("not delivered — the session may be gone") and the buttons re-enable. The ack
   contract (bind/post only) is honored: the card never treats the ack as the answer.
5. **The C-side re-entry rule honored by construction:** `sa_client_ask_reply` refuses inside an
   events callback; Dart replies originate from a BUTTON TAP (main isolate → op isolate), never
   from a C callback. A documented comment pins this at the op.

## 2. The FFI additions (the standing op-addition pattern)

`sa_ffi.dart`: the lookup pair for `sa_client_ask_reply` (the C ABI's own signature; strings
copied to native by the runner like prompt's/interrupt's) + the `SaFfiApi` interface method +
the callback typedef beside the interrupt typedef (:232-277). `sa_client_binding.dart`: the new
`SaOpCommand` kind (:252+), the `_SaOpKind` enum entry (:726) with its `refusal()` shape (:742),
the runner case in `handle` (:438-478), the `NativeCallable.listener` + the `_on…` chain-link
handler, and `SaClientNative.askReply(...)` (:840 area). The op's completion rides the standing
`_PendingOp` serial/dedup machinery verbatim (no new dispose-chain shapes).

## 3. The card (`flutter/lib/views/chat/`)

- Rendered by `MessageCanvas` from the PendingAskStore — the pending ask is a special row at the
  canvas's end (never mid-history; history re-renders don't resurrect an answered ask because the
  store holds only the IN-FLIGHT one).
- Header + question text; the plan block under it via the existing `flutter_markdown`
  (engine-side budget-capped — render verbatim).
- Option buttons: tap → `askReply(..., 0, label)` (the LABEL is the value; an indexed pick is not
  needed — the label rides verbatim).
- Free-text: a one-line input + send affordance ("type a different answer"); posting uses the
  typed value.
- Reject: the card shows its own reject affordance when the ask's options are EMPTY; an options
  list that merely lacks a reject-shaped entry is not detected (recorded). The reject posts
  `askReply(..., 1, value)` (empty value → the engine's default refusal wording).
- In-flight: buttons disable + a spinner line until the op resolves; then either the card
  clears (delivered) or the error line + re-enabled buttons (not delivered).
- The silent-drop corner (pinned): a DELIVERED ack re-enables the buttons immediately, but the
  card clears only when the `ask.reply` record lands (the events stream is the truth, §1's ack
  contract). A stale reply the engine dropped silently (ack delivered, no record ever) leaves
  the card re-enabled — retryable, honest to the bind/post contract; no retry-count machinery.

## 4. Mock mode

`MockChatService` (flutter/lib/data/mock/mock_services.dart:105) gains a scripted ask: a magic
phrase ("plan") makes its reply flow emit a plan-gate-shaped ask (question "Approve this plan?",
options `["Approve", "Reject"]`, a small plan text) through the SAME store path; answering
(`answerAsk`) writes the assistant reply straight into the session store — no typing beat (the
send stream already ended with the ask; `value` rides the interface, but the mock's scripted
reply text is const). Mock implements the same PendingAskStore contract so
view tests drive the card without the C side.

## 5. Tests (the standing bar: `flutter analyze` zero + `flutter test`)

- **`test/ffi/ffi_test.dart`**: the ask-reply op against `FakeSaFfi` — serial/complete/dedup
  ordering + the cb-fail → failed Future (the interrupt op's existing patterns).
- **`test/daemon/ffi_services_test.dart`**: the fold — ask record → store entry; `turn.end{blocked}`
  leaves the turn open (no done/fail); delivered reply → `ask.reply` clears the entry + the
  answer's `msg.append` bubble lands; delivered:false keeps the entry; a second ask record
  (impossible engine-side, but a malicious/garbled stream) replaces honestly, never double-mounts.
- **`test/chat_view_test.dart`**: card render (question/options/plan markdown); option tap → the
  op posts with the label; free-text tap → posts the typed value; reject-without-options;
  in-flight disabling + the not-delivered error line.
- Existing suites stay green (the fold's new cases are additive; standing pins hold).

## 6. Non-goals (recorded, not built)

- Badges/presentation for asks in NON-active sessions (the store holds them keyed by sid; a
  surface for them is a later card).
- A global interrupt button (the app has none today — its own seam).
- The engine-side scope (the wire pair, the park, BYPASS) — SecretAgent's escalation slice.
- Ask history in the canvas (an answered ask is just the reply records — no special archive).
