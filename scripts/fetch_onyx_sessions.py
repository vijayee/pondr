"""Fetch Onyx chat sessions -> sessions.jsonl (STRM Phase 2d v2 substrate).

The v2 graduation head is trained on ``later_needed`` labels derived from
multi-turn sessions: after a WM ring slot was compressed out, did a LATER turn
in the same session reference its content (the "would-have-been-needed"
signal)? Onyx (the user's self-hosted chat server) is the substrate -- its
sessions are real multi-turn conversations with provenance. This script pulls
them to ``data/training/strm_graduation/sessions.jsonl`` (gitignored) for
``scripts/generate_graduation_labels.py`` (Step 5) to label.

SECURITY -- the Onyx API key is a SECRET. Env-var auth ONLY:
  ONYX_BASE_URL  e.g. http://192.168.1.198  -> API base = ``$ONYX_BASE_URL/api``
  ONYX_API_KEY   Bearer token. REFUSE to run if unset. NEVER written to disk,
                 NEVER logged: the Authorization header is built but never
                 printed; HTTP error messages scrub it (only the status, URL,
                 and response body are printed -- the URL carries no secret).

Endpoints (Onyx API, confirmed against docs.onyx.app):
  GET /chat/search?query=&page=&page_size=
      -> {groups:[{title,chats:[ChatSessionSummary{id,name,time_created,...}]}],
          has_more, next_page}
  GET /chat/get-chat-session/{session_id}?is_shared=false&include_deleted=false
      -> ChatSessionDetailResponse {chat_session_id, description, time_created,
          messages:[ChatMessageDetail{message_id, parent_message, message,
          message_type (system|user|assistant|tool_call|tool_call_response),
          time_sent, citations, error, ...}], ...}

Algorithm: list sessions (paginate via ``page`` until ``has_more`` is false;
optional ``--query`` filters, ``--limit`` caps), fetch each session's detail,
extract the message thread (sorted by ``time_sent`` -- chronological, the order
that matters for "a LATER turn referenced this"; ``parent_message`` preserved
so downstream can reconstruct the message tree if needed), emit one JSONL
record per session: ``{session_id, name, time_created, has_error, messages:[{
message_id, parent_message, role, text, time_sent, citations}]}``. By default
sessions with any errored message are skipped (not clean training substrate);
``--include-failed`` keeps them (still marked ``has_error``).

The HTTP seam is a single ``get_json(url, params)`` callable (DI) so the tests
mock HTTP without a real key or the ``responses`` dep -- ``main()`` builds the
real callable (a ``requests.Session`` + Bearer header), tests pass a fake.

Usage:
    ONYX_BASE_URL=http://192.168.1.198 ONYX_API_KEY=*** \\
        python scripts/fetch_onyx_sessions.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

DEFAULT_OUTPUT = "data/training/strm_graduation/sessions.jsonl"


# ── env (SECRET: the key is never written to disk or logged) ──

def require_env() -> tuple[str, str]:
    """Read ``ONYX_BASE_URL`` + ``ONYX_API_KEY`` from env; refuse if missing.

    The API key is a SECRET -- it is returned to the caller (to build the
    Bearer header) but NEVER printed, NEVER written to the output JSONL, NEVER
    logged. A missing key / base URL is a hard error (exit 1 from ``main``),
    not a silent fallback to a public endpoint.
    """
    base = os.environ.get("ONYX_BASE_URL", "").strip().rstrip("/")
    key = os.environ.get("ONYX_API_KEY", "").strip()
    if not base:
        raise RuntimeError(
            "ONYX_BASE_URL is not set (e.g. http://192.168.1.198). Export it "
            "and retry -- the Onyx API base is $ONYX_BASE_URL/api."
        )
    if not key:
        raise RuntimeError(
            "ONYX_API_KEY is not set. Export it (the Onyx Bearer token) and "
            "retry. The key is read from env ONLY -- it is never written to "
            "disk and never logged."
        )
    return base, key


# ── HTTP seam (DI: main() builds the real callable, tests pass a fake) ──

def make_get_json(base_url: str, api_key: str) -> Callable[[str, dict], dict]:
    """Build the real ``get_json(url, params)`` over a ``requests.Session``.

    The Bearer header is attached to the session ONCE (not per call) so the
    key is handled in exactly one place. ``url`` is the path AFTER the API base
    (e.g. ``/chat/search``); the base + url join happens here. Raises on a non-
    2xx with a message that scrubs the headers (only status + url + body print
    -- the url carries no secret).
    """
    import requests  # local import: scripts need not force a top-level dep

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    })
    api_base = base_url.rstrip("/") + "/api"

    def get_json(url: str, params: Optional[dict] = None) -> dict:
        full = api_base + "/" + url.lstrip("/")
        resp = session.get(full, params=params or {}, timeout=30)
        if not resp.ok:
            # Scrub headers from the error -- the Authorization header is a
            # secret. Print only status + url + body (the url carries no key).
            body = resp.text[:500]
            raise RuntimeError(
                f"Onyx API {resp.status_code} at {url} (params={params}): {body}"
            )
        return resp.json()

    return get_json


# ── listing + detail ──

def list_sessions(
    get_json: Callable[[str, dict], dict],
    query: Optional[str] = None,
    page_size: int = 50,
    limit: Optional[int] = None,
) -> list[dict]:
    """Page through ``/chat/search`` until ``has_more`` is false.

    Returns the flat list of ``ChatSessionSummary`` dicts (the ``chats`` across
    all groups, all pages). Stops when ``has_more`` is false, when ``next_page``
    is null/absent, or when ``limit`` is reached. ``query`` filters server-side
    (None -> list all).
    """
    out: list[dict] = []
    page = 1
    while True:
        params = {"page": page, "page_size": page_size}
        if query:
            params["query"] = query
        data = get_json("/chat/search", params)
        for group in data.get("groups", []) or []:
            for chat in group.get("chats", []) or []:
                out.append(chat)
                if limit is not None and len(out) >= limit:
                    return out
        if not data.get("has_more"):
            return out
        nxt = data.get("next_page")
        if nxt is None:
            return out
        page = int(nxt)
        # belt-and-suspenders: a server that returns has_more=true with a
        # next_page equal to the current page would loop forever -- advance.
        if page <= int(params["page"]):
            page = int(params["page"]) + 1
    return out


def fetch_session_detail(
    get_json: Callable[[str, dict], dict],
    session_id: str,
) -> dict:
    """``GET /chat/get-chat-session/{session_id}`` -> ChatSessionDetailResponse."""
    return get_json(f"/chat/get-chat-session/{session_id}",
                    {"is_shared": False, "include_deleted": False})


# ── message-thread extraction ──

def reconstruct_messages(messages: list[dict]) -> list[dict]:
    """Map ChatMessageDetail -> the emit shape, sorted chronologically.

    Sorted by ``(time_sent, message_id)`` -- the chronological order that
    matters for "a LATER turn referenced this slot" (the graduation label).
    ``parent_message`` is preserved so downstream can reconstruct the message
    tree (Onyx messages branch: a user turn may have several assistant
    responses). ``role`` is the raw ``message_type`` (system/user/assistant/
    tool_call/tool_call_response) so the labeler can filter to user/assistant
    turns. ``citations`` is the int->string map verbatim (or null).
    """
    def sort_key(m) -> tuple:
        if not isinstance(m, dict):
            # non-dict items sort last (they are filtered out below); a stable
            # high key keeps them from crashing sorted().
            return ("￿", 0)
        ts = m.get("time_sent") or ""
        mid = m.get("message_id")
        try:
            mid_i = int(mid) if mid is not None else 0
        except (TypeError, ValueError):
            mid_i = 0
        return (ts, mid_i)

    out: list[dict] = []
    if not isinstance(messages, list):
        return out                          # malformed detail -> empty thread, not a crash
    for m in sorted(messages, key=sort_key):
        if not isinstance(m, dict):
            continue
        out.append({
            "message_id": m.get("message_id"),
            "parent_message": m.get("parent_message"),
            "role": m.get("message_type"),
            "text": m.get("message") or "",
            "time_sent": m.get("time_sent"),
            "citations": m.get("citations"),
        })
    return out


def session_has_error(detail: dict) -> bool:
    """True if any message in the session detail carries a non-null ``error``."""
    for m in detail.get("messages", []) or []:
        if m.get("error"):
            return True
    return False


def session_record(summary: dict, detail: dict) -> dict:
    """Build the JSONL record from the search summary + the session detail.

    ``session_id`` from the detail's ``chat_session_id`` (falling back to the
    summary's ``id``); ``name`` from the detail's ``description`` (falling back
    to the summary's ``name``); ``time_created`` from the detail (falling back
    to the summary). ``has_error`` is always present so downstream can filter
    regardless of ``--include-failed``.
    """
    session_id = detail.get("chat_session_id") or summary.get("id")
    name = detail.get("description") or summary.get("name") or session_id
    time_created = detail.get("time_created") or summary.get("time_created")
    return {
        "session_id": session_id,
        "name": name,
        "time_created": time_created,
        "has_error": session_has_error(detail),
        "messages": reconstruct_messages(detail.get("messages", []) or []),
    }


# ── orchestration ──

def fetch_all(
    get_json: Callable[[str, dict], dict],
    query: Optional[str],
    page_size: int,
    limit: Optional[int],
    include_failed: bool,
    verbose: bool = False,
) -> list[dict]:
    """List sessions, fetch each detail, build records. Skips errored sessions
    unless ``include_failed``. Returns the records (one per kept session)."""
    summaries = list_sessions(get_json, query=query, page_size=page_size,
                              limit=limit)
    if verbose:
        print(f"  listed {len(summaries)} sessions", flush=True)
    records: list[dict] = []
    for i, s in enumerate(summaries):
        sid = s.get("id")
        if not sid:
            continue
        try:
            detail = fetch_session_detail(get_json, sid)
            rec = session_record(s, detail)
        except Exception as e:  # noqa: BLE001 - one bad session skips, not aborts
            if verbose:
                print(f"  skip {sid}: {e}", flush=True)
            continue
        if rec["has_error"] and not include_failed:
            if verbose:
                print(f"  skip {sid} (has errored message; --include-failed to keep)",
                      flush=True)
            continue
        records.append(rec)
        if verbose and (i + 1) % 10 == 0:
            print(f"  fetched {i + 1}/{len(summaries)} sessions", flush=True)
    return records


def write_jsonl(records: list[dict], output: str) -> None:
    """Append-safe JSONL write (one record per line, ensure_ascii=False)."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fetch Onyx chat sessions -> sessions.jsonl (STRM 2d v2 substrate)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="JSONL output path (default: data/training/strm_graduation/sessions.jsonl)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on the number of sessions to fetch (default: all)")
    p.add_argument("--query", default=None,
                   help="server-side search filter (default: list all sessions)")
    p.add_argument("--page-size", type=int, default=50,
                   help="chat/search page_size (default 50)")
    p.add_argument("--include-failed", action="store_true",
                   help="keep sessions with errored messages (default: skip them)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        base, _key = require_env()      # _key is a SECRET -- never printed
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.verbose:
        # print the BASE (not the key) so the user sees which server is hit
        print(f"Onyx base: {base}/api", flush=True)

    get_json = make_get_json(base, _key)
    try:
        records = fetch_all(
            get_json, query=args.query, page_size=args.page_size,
            limit=args.limit, include_failed=args.include_failed,
            verbose=args.verbose,
        )
    except Exception as e:  # noqa: BLE001 - top-level: print a scrubbed message
        # The error from get_json scrubs the Authorization header; surface it.
        print(f"ERROR: fetch failed -- {e}", file=sys.stderr)
        return 1

    write_jsonl(records, args.output)
    n_msg = sum(len(r["messages"]) for r in records)
    print(f"DONE. {len(records)} sessions, {n_msg} messages -> {args.output}",
          flush=True)
    print(f"  Next: python scripts/generate_graduation_labels.py "
          f"--sessions {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())