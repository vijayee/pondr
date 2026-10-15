"""Inventory the user's Onyx chat sessions (cursor pagination), then pull
each session's messages into the mineable store.

  listing: GET /api/chat/get-user-chat-sessions?page_size=100&before=<ts>
  cursor   = the OLDEST time_updated of the last page (newest-first)
  messages: GET /api/chat/get-chat-session/{id}

Auth: cookie (fastapiusersauth) from a gitignored file. Output saved
incrementally so an interrupted run resumes.
Outputs:
  data/_scratch/icm/capture/onyx_inventory.jsonl   (session list)
  data/_scratch/icm/capture/onyx_chats.jsonl        (full sessions)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

OUTDIR = os.path.join("data", "_scratch", "icm", "capture")
INV = os.path.join(OUTDIR, "onyx_inventory.jsonl")
CHATS = os.path.join(OUTDIR, "onyx_chats.jsonl")
# base URL + cookie NAME live in a gitignored config; nothing identifying
# is ever tracked (user directive: personal chats never committed)
CONFIG = os.path.join("data", "_scratch", "onyx_config.json")
COOKIE = os.path.join("data", "_scratch", "onyx_cookie.txt")


def _cfg():
    return json.load(io.open(CONFIG, encoding="utf-8"))


def cookie() -> str:
    v = open(COOKIE, encoding="utf-8").read().strip()
    if not v:
        raise SystemExit("cookie file empty (data/_scratch/onyx_cookie.txt)")
    return _cfg().get("cookie_name", "fastapiusersauth") + "=" + v


def get(path: str, timeout: int = 120):
    req = urllib.request.Request(_cfg()["base"] + path,
                                 headers={"Cookie": cookie(), "Accept": "*/*",
                                          "User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def inventory(max_pages: int = 50):
    done_before = set()
    if os.path.exists(INV):
        for l in io.open(INV, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("_last_page"):
                    done_before.add(r["_last_page"])
    sessions, page = [], 0
    before = None
    with io.open(INV, "a", encoding="utf-8", newline="\n") as f:
        while page < max_pages:
            q = "/api/chat/get-user-chat-sessions?page_size=100&only_non_project_chats=false"
            if before:
                q += f"&before={urllib.request.quote(before)}"
            p = get(q)
            items = p.get("sessions") or []
            if not items:
                break
            for s in items:
                f.write(json.dumps({"id": s["id"], "name": s.get("name"),
                                    "created": s.get("time_created"),
                                    "updated": s.get("time_updated")}) + "\n")
            sessions.extend(items)
            before = min(s.get("time_updated", "") for s in items)
            print(f"page {page}: +{len(items)} (oldest: {before[:19]}, "
                  f"total {len(sessions)})", flush=True)
            if not p.get("has_more"):
                break
            before_key = before
            before = before_key
            page += 1
            time.sleep(0.4)
    return sessions


def pull_chats(limit: int = 0):
    ids = []
    for l in io.open(INV, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if "id" in r and r["id"] not in ids:
                ids.append(r["id"])
    done = set()
    if os.path.exists(CHATS):
        for l in io.open(CHATS, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["chat_session"]["id"])
                except Exception:
                    pass
    todo = [i for i in ids if i not in done][:limit] if limit else [i for i in ids if i not in done]
    print(f"[pull] {len(todo)} chats to fetch", flush=True)
    ok = fail = 0
    with io.open(CHATS, "a", encoding="utf-8", newline="\n") as f:
        for k, sid in enumerate(todo):
            try:
                s = get(f"/api/chat/get-chat-session/{sid}")
                f.write(json.dumps({"chat_session": s}) + "\n")
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  [{k}] FAIL {sid[:8]}: {str(e)[:60]}", flush=True)
            if k % 20 == 0:
                print(f"  ...{k}/{len(todo)} (ok {ok}, fail {fail})", flush=True)
            time.sleep(0.35)
    print(f"[pull] done: ok {ok}, fail {fail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-limit", type=int, default=0)
    ap.add_argument("--inventory-only", action="store_true")
    args = ap.parse_args()
    sessions = inventory()
    print(f"[inventory] {len(sessions)} sessions listed")
    if not args.inventory_only:
        pull_chats(limit=args.pull_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())