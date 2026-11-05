/// The pending-ask card's state (the escalation slice's Pondr half —
/// `docs/ask-ux-spec.md` §1): one IN-FLIGHT owner-surface ask per session.
///
/// The engine's law carries the shape: at most ONE parked ask per frame —
/// the fold's `ask` record upserts (last-writer-wins per sid; a malicious
/// or garbled stream replaces honestly, never double-mounts) and only the
/// `ask.reply` record clears. An answered ask is just the reply records —
/// the store holds no archive.
///
/// The provider lives beside the store (the `chat_state.dart` idiom: stores
/// register their own `NotifierProvider`; `bindings.dart` stays a
/// service-swap file). The stream it subscribes is the mode's one swap —
/// the daemon's record fold and the mock's scripted ask ride the SAME path,
/// so the view tests drive the card without the C side.
library;

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'bindings.dart'; // chatServiceProvider (the mode's one swap)

/// One in-flight ask (the escalation slice's parked ask): the card's
/// question, its option labels, and the plan block shown under the
/// question (null when the ask carried none).
final class PendingAsk {
  const PendingAsk({
    required this.askId,
    required this.question,
    required this.options,
    this.plan,
  });

  final String askId;
  final String question;
  final List<String> options;
  final String? plan;
}

/// The ask stream's upsert — the fold's `ask` record (the parked frame).
final class AskUpsert extends AskEvent {
  const AskUpsert(this.sid, this.ask);

  final String sid;
  final PendingAsk ask;
}

/// The ask stream's clear — the fold's `ask.reply` record (the answer
/// entered the frame's mailbox; the record, not the reply op's ack, is the
/// card's truth).
final class AskCleared extends AskEvent {
  const AskCleared(this.sid);

  final String sid;
}

/// The ask events' common base — the interface's `Stream<Object>` carries
/// the union; the store switches on the runtime type.
sealed class AskEvent {
  const AskEvent();
}

/// The card's single source of truth, keyed by sid.
///
/// Reactivity: the asks stream is a stream of CHANGE EVENTS (the other
/// stores' idiom) — an [AskUpsert] parks the sid's ask, an [AskCleared]
/// drops it. [PendingAsk] has no identity-based equality on purpose: the
/// REPLACEMENT ask must be seen as a change even when its fields match the
/// old one's (fresh list identity each upsert). The stream is broadcast and
/// UNBUFFERED — an ask record landing before anything watches the store is
/// dropped; recovery is an app RESTART (a fresh fold state) or the next
/// turn's stream — a same-run re-subscribe replays nothing (the fold's
/// (sid, seq) dedupe never clears, so a replay re-folds nothing).
class PendingAskStore extends Notifier<Map<String, PendingAsk>> {
  @override
  Map<String, PendingAsk> build() {
    final chat = ref.watch(chatServiceProvider);
    final StreamSubscription<Object> sub = chat.asks().listen((Object event) {
      state = switch (event) {
        AskUpsert(:final sid, :final ask) => {...state, sid: ask},
        AskCleared(:final sid) => {...state}..remove(sid),
        // The stream is typed Object (the event models live beside the
        // store, not in the interface); only the ask events mutate.
        _ => state,
      };
    });
    ref.onDispose(sub.cancel);
    return const {};
  }

  /// The sid's in-flight ask (null = nothing parked; the card never rides).
  PendingAsk? forSid(String sid) => state[sid];
}

final pendingAskProvider =
    NotifierProvider<PendingAskStore, Map<String, PendingAsk>>(
      PendingAskStore.new,
    );
