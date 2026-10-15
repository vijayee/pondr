"""Mine Onyx chats (scripts/icm/onyx_inventory.py's onyx_chats.jsonl) into
capture-shaped records for the agent-intent corpus.

The Onyx record's own structure gives us the hierarchy FOR FREE:
  goal      = the chat (its name + the FIRST user message)
  steps     = each user message (real intent-bearing inputs)
  retrieval = each ASSISTANT message's context_docs (Onyx's own retrieval:
              query = the user message that prompted it, retrieved = the
              document links + blurbs Onyx actually pulled). This is the
              agent-native retrieval event the transcripts can't give.
  outcome   = current_feedback == "like" on assistant messages (the ONLY
              organic utility label; 11 exist). Everything else stays
              unlabeled (no fabrication).

Usage:  python scripts/icm/mine_onyx_chats.py
Output: data/_scratch/icm/capture/onyx_mined.jsonl (+ a summary print)
"""

from __future__ import annotations

import io
import json
import os
import sys

IN = os.path.join("data", "_scratch", "icm", "capture", "onyx_chats.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "capture", "onyx_mined.jsonl")


def main() -> int:
    rows = [json.loads(l) for l in io.open(IN, encoding="utf-8") if l.strip()]
    n_chats = n_steps = n_retr = n_out = n_fb = 0
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        def emit(r):
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")
        for row in rows:
            cs = row["chat_session"]
            msgs = cs.get("messages") or []
            cid = cs.get("id") or cs.get("chat_session_id")
            first_user = next((m for m in msgs if m.get("message_type") == "user"), None)
            goal_text = (first_user.get("message") if first_user else
                         cs.get("name") or "unnamed chat")
            emit({"kind": "icapture", "level": "goal", "parent": None,
                  "id": f"oc-{cid}", "text": str(goal_text)[:400],
                  "branches": {"desired": None},
                  "declared_at": first_user.get("time_sent") if first_user else None,
                  "label_basis": "organic-feedback (mined)"})
            n_chats += 1
            last_user = None
            for m in msgs:
                mt = m.get("message_type")
                if mt == "user":
                    last_user = m
                    emit({"kind": "icapture", "level": "step", "parent": f"oc-{cid}",
                          "id": f"oc-{cid}-m{m.get('message_id')}",
                          "text": str(m.get("message", ""))[:400],
                          "branches": {"desired": None},
                          "declared_at": m.get("time_sent"),
                          "label_basis": "organic-feedback (mined)"})
                    n_steps += 1
                elif mt == "assistant":
                    docs = m.get("context_docs") or []
                    if docs and last_user is not None:
                        rid = f"oc-{cid}-retr{m.get('message_id')}"
                        emit({"kind": "icapture", "level": "retrieval",
                              "parent": f"oc-{cid}-m{last_user.get('message_id')}",
                              "id": rid,
                              "query": str(last_user.get("message", ""))[:300],
                              "retrieved": [d.get("link") or d.get("document_id", "?")
                                            for d in docs],
                              "declared_at": m.get("time_sent"),
                              "label_basis": "organic-feedback (mined)"})
                        n_retr += 1
                    if m.get("current_feedback") == "like":
                        emit({"kind": "icapture", "level": "outcome",
                              "parent": f"oc-{cid}-retr{m.get('message_id')}"
                              if docs and last_user is not None
                              else f"oc-{cid}-m{last_user.get('message_id')}"
                              if last_user else f"oc-{cid}",
                              "id": f"oc-{cid}-out{m.get('message_id')}",
                              "observed": "desired",
                              "note": "user gave a like (organic feedback)",
                              "declared_at": m.get("time_sent"),
                              "label_basis": "organic-feedback"})
                        n_out += 1; n_fb += 1
    print(f"[mined] {n_chats} chats -> {n_steps} steps, {n_retr} retrieval events, "
          f"{n_out} outcome labels ({n_fb} organic likes)")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())