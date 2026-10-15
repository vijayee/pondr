"""Mine Onyx chats (v2, user-graded demarcation 2026-10-15) into
capture-shaped records.

Rules per the user's grading of the OCR-chat sample:
  prompt      = the USER's first message (the origin; not assumed the goal)
  goal        = the AGENT's first response (its intent-in-pursuit is revealed
                by its own first reply); goals may SHIFT mid-conversation
  goal-refine = the user's REFUTE/REDIRECT messages (the strongest signal
                class the user named) start a goal refinement lineage
  steps       = every other user message, classified (metadata only):
                criteria | enlightenment | report | neutral | noise
  retrieval   = each assistant message's context_docs (Onyx's own retrieval
                trace: query = the prompting user message, docs = what it
                pulled, with citations).
  outcomes    = current_feedback thumbs ONLY (organic). Class patterns are
                metadata, NEVER labels (the valence law).
LAWS: no stated labels; label_basis on every scored row; noisy rows marked
not dropped.
Output: data/_scratch/icm/capture/onyx_mined.jsonl
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

IN = os.path.join("data", "_scratch", "icm", "capture", "onyx_chats.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "capture", "onyx_mined.jsonl")

REFUTE = re.compile(r"\b(no,|nope|i disagree|that's wrong|actually, no|i mean |"
                    r"you misunderstand|no i mean|not what|stop |"
                    r"failed miserably|that's not|i'm not (asking|saying)|don't)", re.I)
REDIRECT = re.compile(r"\b(instead|rather|switch|pivot|let's take|focus on |"
                      r"i want to think about|new direction|what if we|how about)", re.I)
CRITERIA = re.compile(r"\b(must|should|need to|verify|success (is|would be)|"
                      r"(i )?expect|criteria|so that it|require)", re.I)
ENLIGHTEN = re.compile(r"\b(i understand|i see|that makes sense|aha|"
                       r"(it )?feels right|this is the first|now i (get|see)|"
                       r"i like (that|this)|enlighten)", re.I)
REPORT = re.compile(r"^(\s*(the (verdict|smoke|organic|judge|labels|channel|payoff|weak|"
                    r"adversarial|pre-registered|reality)|E0 is|v1 lands|"
                    r"everything landed|both checks))")
WEIGHT = {"refute/redirect": 3, "criteria": 2, "enlightenment": 2,
          "report": 0, "neutral": 1, "noise": 0}


def classify_user(text: str) -> str:
    t = text.strip()
    if not t or t in {"+", "-", "ok", "OK"}:
        return "noise"
    if REFUTE.search(t) or REDIRECT.search(t):
        return "refute/redirect"
    if CRITERIA.search(t):
        return "criteria"
    if ENLIGHTEN.search(t):
        return "enlightenment"
    if REPORT.match(t):
        return "report"
    return "neutral"


def main() -> int:
    rows = [json.loads(l) for l in io.open(IN, encoding="utf-8") if l.strip()]
    stats = {"chats": 0, "prompts": 0, "goals": 0, "goal_shifts": 0, "steps": 0,
             "retrievals": 0, "outcomes": 0, "likes": 0}
    classes = {}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        def emit(r):
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")
        for row in rows:
            cs = row["chat_session"]
            msgs = cs.get("messages") or []
            cid = cs.get("id") or cs.get("chat_session_id")
            gid = f"oc-{cid}"
            users = [m for m in msgs if m.get("message_type") == "user"]
            assists = [m for m in msgs if m.get("message_type") == "assistant"]
            first_user = users[0] if users else None
            if first_user is not None:
                emit({"kind": "icapture", "level": "prompt", "parent": None,
                      "id": f"{gid}-prompt",
                      "text": str(first_user.get("message", ""))[:400],
                      "declared_at": first_user.get("time_sent"),
                      "label_basis": "organic-feedback"})
                stats["prompts"] += 1
            goal_text = (str(assists[0].get("message", ""))[:400] if assists
                         else str(first_user.get("message", ""))[:400] if first_user
                         else "unnamed chat")
            emit({"kind": "icapture", "level": "goal",
                  "parent": f"{gid}-prompt" if first_user else None,
                  "id": gid, "text": goal_text,
                  "declared_at": assists[0].get("time_sent") if assists else None,
                  "label_basis": "organic-feedback"})
            stats["chats"] += 1; stats["goals"] += 1
            goal_version = 0
            last_user = None
            for m in msgs:
                mt = m.get("message_type")
                if mt == "user":
                    last_user = m
                    cls = classify_user(str(m.get("message", "")))
                    classes[cls] = classes.get(cls, 0) + 1
                    if cls == "refute/redirect":
                        goal_version += 1
                        emit({"kind": "icapture", "level": "goal", "parent": gid,
                              "id": f"{gid}-refine{goal_version}",
                              "text": str(m.get("message", ""))[:400],
                              "signal_class": cls,
                              "declared_at": m.get("time_sent"),
                              "label_basis": "organic-feedback"})
                        stats["goal_shifts"] += 1
                    emit_level = "report" if cls == "report" else "step"
                    emit({"kind": "icapture", "level": emit_level, "parent": gid,
                          "id": f"{gid}-m{m.get('message_id')}",
                          "text": str(m.get("message", ""))[:400],
                          "signal_class": cls,
                          "declared_at": m.get("time_sent"),
                          "label_basis": "organic-feedback"})
                    stats["steps"] += 1
                elif mt == "assistant":
                    docs = m.get("context_docs") or []
                    rid = f"{gid}-retr{m.get('message_id')}"
                    if docs and last_user is not None:
                        emit({"kind": "icapture", "level": "retrieval",
                              "parent": f"{gid}-m{last_user.get('message_id')}",
                              "id": rid,
                              "query": str(last_user.get("message", ""))[:300],
                              "retrieved": [d.get("link") or d.get("document_id", "?")
                                            for d in docs],
                              "declared_at": m.get("time_sent"),
                              "label_basis": "organic-feedback"})
                        stats["retrievals"] += 1
                    if m.get("current_feedback") == "like":
                        emit({"kind": "icapture", "level": "outcome",
                              "parent": rid if docs and last_user is not None
                              else (f"{gid}-m{last_user.get('message_id')}"
                                    if last_user else gid),
                              "id": f"{gid}-out{m.get('message_id')}",
                              "observed": "desired",
                              "note": "organic like",
                              "declared_at": m.get("time_sent"),
                              "label_basis": "organic-feedback"})
                        stats["outcomes"] += 1; stats["likes"] += 1
    print("[mined v2]", json.dumps(stats))
    print("  user-class histogram:", json.dumps(classes))
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())